"""
Threading and concurrency utilities for XPCS Viewer.

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
from .batch_bayesian_coordinator import BatchBayesianCoordinator
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
    # Batch Bayesian coordinator
    "BatchBayesianCoordinator",
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
    "TwotimePlotWorker",
    "UnifiedTask",
    "UnifiedThreadingManager",
    "WorkerManager",
    "WorkerSignals",
    "get_unified_threading_manager",
    "shutdown_unified_threading",
]
