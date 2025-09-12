"""
Utilities package for XPCS Toolkit.

This package contains utility modules for performance profiling,
optimization helpers, logging infrastructure, and other support functions.
"""

# Import logging utilities - these should always be available
try:
    from .logging_config import (
        get_logger,
        get_logging_config,
        set_log_level,
        get_log_file_path,
        log_system_info,
        setup_exception_logging,
        setup_logging,
        initialize_logging,
    )
    from .log_formatters import (
        ColoredConsoleFormatter,
        StructuredFileFormatter,
        JSONFormatter,
        PerformanceFormatter,
        create_formatter,
    )

    _logging_available = True
    _logging_exports = [
        "get_logger",
        "get_logging_config",
        "set_log_level",
        "get_log_file_path",
        "log_system_info",
        "setup_exception_logging",
        "setup_logging",
        "initialize_logging",
        "ColoredConsoleFormatter",
        "StructuredFileFormatter",
        "JSONFormatter",
        "PerformanceFormatter",
        "create_formatter",
    ]
except ImportError as e:
    print(f"Warning: Logging utilities not available: {e}")
    _logging_available = False
    _logging_exports = []

# Import performance profiler - optional dependency
try:
    from .performance_profiler import (
        PerformanceProfiler,
        global_profiler,
        profile_algorithm,
        profile_block,
        benchmark_function,
        memory_usage_profiler,
    )

    _profiler_available = True
    _profiler_exports = [
        "PerformanceProfiler",
        "global_profiler",
        "profile_algorithm",
        "profile_block",
        "benchmark_function",
        "memory_usage_profiler",
    ]
except ImportError as e:
    print(f"Warning: Performance profiling not available: {e}")
    _profiler_available = False
    _profiler_exports = []

# Import advanced caching system - comprehensive performance optimization
try:
    from .advanced_cache import (
        get_global_cache,
        MultiLevelCache,
        CacheLevel,
        EvictionPolicy,
        cache_computation,
    )

    from .computation_cache import (
        get_computation_cache,
        ComputationCache,
        G2FitResult,
        SAXSResult,
        TwoTimeResult,
        cache_g2_fitting,
        cache_saxs_analysis,
        cache_twotime_correlation,
    )

    from .metadata_cache import (
        get_metadata_cache,
        MetadataCache,
        FileMetadata,
        QMapData,
        cache_file_metadata,
    )

    from .adaptive_memory import (
        get_adaptive_memory_manager,
        AdaptiveMemoryManager,
        MemoryStrategy,
        UsagePattern,
        smart_cache_decorator,
    )

    from .cache_monitor import (
        get_cache_monitor,
        CacheMonitor,
        PerformanceAlert,
        AlertLevel,
        setup_cache_monitoring_gui_integration,
    )

    from .enhanced_xpcs_file import (
        CachedXpcsFileMixin,
        enhance_xpcs_file_with_caching,
        SmartXpcsFileManager,
    )

    from .enhanced_viewer_kernel import (
        CachedViewerKernelMixin,
        enhance_viewer_kernel_with_caching,
        SmartViewerKernelManager,
        get_global_viewer_manager,
    )

    _caching_available = True
    _caching_exports = [
        "get_global_cache",
        "MultiLevelCache",
        "CacheLevel",
        "EvictionPolicy",
        "cache_computation",
        "get_computation_cache",
        "ComputationCache",
        "G2FitResult",
        "SAXSResult",
        "TwoTimeResult",
        "cache_g2_fitting",
        "cache_saxs_analysis",
        "cache_twotime_correlation",
        "get_metadata_cache",
        "MetadataCache",
        "FileMetadata",
        "QMapData",
        "cache_file_metadata",
        "get_adaptive_memory_manager",
        "AdaptiveMemoryManager",
        "MemoryStrategy",
        "UsagePattern",
        "smart_cache_decorator",
        "get_cache_monitor",
        "CacheMonitor",
        "PerformanceAlert",
        "AlertLevel",
        "setup_cache_monitoring_gui_integration",
        "CachedXpcsFileMixin",
        "enhance_xpcs_file_with_caching",
        "SmartXpcsFileManager",
        "CachedViewerKernelMixin",
        "enhance_viewer_kernel_with_caching",
        "SmartViewerKernelManager",
        "get_global_viewer_manager",
    ]

    # Convenience functions for easy setup
    def setup_advanced_caching(
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
        enable_monitoring: bool = True,
        enable_gui_integration: bool = False,
    ):
        """
        Setup the complete advanced caching system with recommended defaults.

        Parameters
        ----------
        strategy : MemoryStrategy
            Memory management strategy (CONSERVATIVE, BALANCED, AGGRESSIVE)
        enable_monitoring : bool
            Whether to enable cache monitoring and performance alerts
        enable_gui_integration : bool
            Whether to setup GUI integration for cache monitoring

        Returns
        -------
        dict
            Dictionary containing references to all cache components
        """
        # Initialize all cache components
        advanced_cache = get_global_cache()
        computation_cache = get_computation_cache()
        metadata_cache = get_metadata_cache()
        memory_manager = get_adaptive_memory_manager(strategy=strategy)

        components = {
            "advanced_cache": advanced_cache,
            "computation_cache": computation_cache,
            "metadata_cache": metadata_cache,
            "memory_manager": memory_manager,
        }

        if enable_monitoring:
            monitor = get_cache_monitor()
            components["monitor"] = monitor

            if enable_gui_integration:
                setup_cache_monitoring_gui_integration()

        return components

    def get_cache_statistics():
        """
        Get comprehensive statistics from all cache components.

        Returns
        -------
        dict
            Combined statistics from all cache systems
        """
        stats = {}

        try:
            stats["advanced_cache"] = get_global_cache().get_stats()
        except Exception as e:
            stats["advanced_cache"] = {"error": str(e)}

        try:
            stats["computation_cache"] = get_computation_cache().get_computation_stats()
        except Exception as e:
            stats["computation_cache"] = {"error": str(e)}

        try:
            stats["metadata_cache"] = get_metadata_cache().get_cache_statistics()
        except Exception as e:
            stats["metadata_cache"] = {"error": str(e)}

        try:
            stats["memory_manager"] = (
                get_adaptive_memory_manager().get_performance_stats()
            )
        except Exception as e:
            stats["memory_manager"] = {"error": str(e)}

        try:
            stats["monitor"] = get_cache_monitor().get_performance_summary()
        except Exception as e:
            stats["monitor"] = {"error": str(e)}

        return stats

    def clear_all_caches(cache_types: list = None):
        """
        Clear all caches in the system.

        Parameters
        ----------
        cache_types : list, optional
            List of cache types to clear. If None, clears all.
            Options: ['advanced', 'computation', 'metadata']
        """
        if cache_types is None:
            cache_types = ["advanced", "computation", "metadata"]

        if "advanced" in cache_types:
            try:
                get_global_cache().clear()
            except Exception as e:
                if _logging_available:
                    logger = get_logger(__name__)
                    logger.error(f"Error clearing advanced cache: {e}")
                else:
                    print(f"Error clearing advanced cache: {e}")

        if "computation" in cache_types:
            try:
                get_computation_cache().cleanup_old_computations(max_age_hours=0.01)
            except Exception as e:
                if _logging_available:
                    logger = get_logger(__name__)
                    logger.error(f"Error clearing computation cache: {e}")
                else:
                    print(f"Error clearing computation cache: {e}")

        if "metadata" in cache_types:
            try:
                metadata_cache = get_metadata_cache()
                metadata_cache.cleanup_expired_metadata(max_age_hours=0.01)
            except Exception as e:
                if _logging_available:
                    logger = get_logger(__name__)
                    logger.error(f"Error clearing metadata cache: {e}")
                else:
                    print(f"Error clearing metadata cache: {e}")

    _caching_exports.extend(
        ["setup_advanced_caching", "get_cache_statistics", "clear_all_caches"]
    )

except ImportError as e:
    print(f"Warning: Advanced caching system not available: {e}")
    _caching_available = False
    _caching_exports = []

# Import memory utilities - core optimization components
try:
    from .memory_utils import (
        MemoryTracker,
        MemoryOptimizer,
        SystemMemoryMonitor,
        memory_profiler,
    )

    _memory_available = True
    _memory_exports = [
        "MemoryTracker",
        "MemoryOptimizer",
        "SystemMemoryMonitor",
        "memory_profiler",
    ]
except ImportError as e:
    print(f"Warning: Memory utilities not available: {e}")
    _memory_available = False
    _memory_exports = []

# Import IO performance utilities
try:
    from .io_performance import (
        get_performance_monitor,
        estimate_hdf5_dataset_size,
        IOPerformanceMonitor,
    )

    _io_performance_available = True
    _io_performance_exports = [
        "get_performance_monitor",
        "estimate_hdf5_dataset_size",
        "IOPerformanceMonitor",
    ]
except ImportError as e:
    print(f"Warning: IO performance utilities not available: {e}")
    _io_performance_available = False
    _io_performance_exports = []

# Import CPU optimization ecosystem - comprehensive optimization management
try:
    from .optimization_ecosystem import (
        get_optimization_ecosystem,
        start_optimization_ecosystem,
        stop_optimization_ecosystem,
        get_ecosystem_status,
        generate_ecosystem_report,
        run_ecosystem_maintenance,
        analyze_ecosystem_performance,
        OptimizationEcosystem,
        EcosystemState,
        EcosystemStatus,
    )

    # Import individual optimization components for direct access
    from .optimization_health_monitor import (
        get_health_monitor,
        OptimizationHealthMonitor,
        HealthCheckResult,
    )

    from .performance_dashboard import (
        create_performance_dashboard,
        PerformanceDashboard,
    )

    from .maintenance_scheduler import (
        get_maintenance_scheduler,
        MaintenanceScheduler,
        MaintenanceTask,
        MaintenanceResult,
    )

    from .alert_system import get_alert_system, AlertSystem, Alert, AlertSeverity

    from .workflow_profiler import get_workflow_profiler, WorkflowProfiler, ProfileStep

    from .cpu_bottleneck_analyzer import (
        get_bottleneck_analyzer,
        CPUBottleneckAnalyzer,
        BottleneckType,
    )

    from .usage_pattern_miner import get_pattern_miner, UsagePatternMiner, UsagePattern

    from .workflow_optimization_report import (
        get_report_generator,
        WorkflowOptimizationReportGenerator,
        ReportFormat,
    )

    _ecosystem_available = True
    _ecosystem_exports = [
        # Main ecosystem interface
        "get_optimization_ecosystem",
        "start_optimization_ecosystem",
        "stop_optimization_ecosystem",
        "get_ecosystem_status",
        "generate_ecosystem_report",
        "run_ecosystem_maintenance",
        "analyze_ecosystem_performance",
        "OptimizationEcosystem",
        "EcosystemState",
        "EcosystemStatus",
        # Subagent 1: Monitoring & Maintenance
        "get_health_monitor",
        "OptimizationHealthMonitor",
        "HealthCheckResult",
        "create_performance_dashboard",
        "PerformanceDashboard",
        "get_maintenance_scheduler",
        "MaintenanceScheduler",
        "MaintenanceTask",
        "MaintenanceResult",
        "get_alert_system",
        "AlertSystem",
        "Alert",
        "AlertSeverity",
        # Subagent 2: Workflow Profiling
        "get_workflow_profiler",
        "WorkflowProfiler",
        "ProfileStep",
        "get_bottleneck_analyzer",
        "CPUBottleneckAnalyzer",
        "BottleneckType",
        "get_pattern_miner",
        "UsagePatternMiner",
        "UsagePattern",
        "get_report_generator",
        "WorkflowOptimizationReportGenerator",
        "ReportFormat",
    ]

    # Convenience functions for easy ecosystem management
    def setup_complete_optimization_ecosystem(
        enable_dashboard: bool = True,
        enable_profiling: bool = True,
        enable_alerts: bool = True,
        profile_all_workflows: bool = False,
    ):
        """
        Setup the complete CPU optimization ecosystem with all subagents.

        This function starts:
        - Subagent 1: Optimization monitoring, health checks, maintenance, alerts
        - Subagent 2: Workflow profiling, bottleneck analysis, usage patterns
        - Subagent 3: Performance testing integration (via ecosystem)

        Parameters
        ----------
        enable_dashboard : bool
            Whether to start the performance dashboard GUI
        enable_profiling : bool
            Whether to enable workflow profiling
        enable_alerts : bool
            Whether to enable alerting system
        profile_all_workflows : bool
            Whether to profile all workflows automatically

        Returns
        -------
        bool
            True if ecosystem started successfully
        """
        return start_optimization_ecosystem(
            enable_dashboard, enable_profiling, enable_alerts, profile_all_workflows
        )

    def get_optimization_status_summary():
        """
        Get a comprehensive optimization status summary.

        Returns
        -------
        dict
            Summary of optimization system status
        """
        status = get_ecosystem_status()
        analysis = analyze_ecosystem_performance()

        return {
            "ecosystem_state": status.state.value,
            "uptime_hours": status.uptime_seconds / 3600.0,
            "performance_score": status.performance_score,
            "ecosystem_health": analysis.get("ecosystem_health", 0.0),
            "active_components": sum(status.components_active.values()),
            "total_components": len(status.components_active),
            "active_alerts": status.active_alerts,
            "critical_issues": status.critical_issues,
            "optimization_opportunities": analysis.get("optimization_opportunities", 0),
            "recommendations": len(analysis.get("recommendations", [])),
        }

    def shutdown_all_optimizations():
        """
        Shutdown all optimization systems gracefully.

        Returns
        -------
        bool
            True if shutdown successful
        """
        return stop_optimization_ecosystem()

    # Legacy compatibility functions
    def start_optimization_monitoring():
        """Legacy compatibility - use setup_complete_optimization_ecosystem instead."""
        return setup_complete_optimization_ecosystem(
            enable_dashboard=False, enable_profiling=False, enable_alerts=True
        )

    def stop_optimization_monitoring():
        """Legacy compatibility - use shutdown_all_optimizations instead."""
        return shutdown_all_optimizations()

    def get_optimization_monitoring_status():
        """Legacy compatibility - use get_optimization_status_summary instead."""
        return get_optimization_status_summary()

    _ecosystem_exports.extend(
        [
            "setup_complete_optimization_ecosystem",
            "get_optimization_status_summary",
            "shutdown_all_optimizations",
            "start_optimization_monitoring",  # Legacy
            "stop_optimization_monitoring",  # Legacy
            "get_optimization_monitoring_status",  # Legacy
        ]
    )

except ImportError as e:
    print(f"Warning: CPU optimization ecosystem not available: {e}")
    _ecosystem_available = False
    _ecosystem_exports = []

# Build the __all__ list based on available components
__all__ = (
    _logging_exports
    + _profiler_exports
    + _caching_exports
    + _memory_exports
    + _io_performance_exports
    + _ecosystem_exports
)
